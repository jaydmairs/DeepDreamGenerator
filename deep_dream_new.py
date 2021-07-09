import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import urllib.request
import os
import zipfile
import time
from tkinter import *
import math

#matplotlib.use('Qt4Agg', warn=False)

def main():
    #Step 1 - download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = './data/'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
  
    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
  
    model_fn = 'tensorflow_inception_graph.pb'
    
    #Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)
    #read in graph file
    # with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
    with tf.compat.v2.io.gfile.GFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.compat.v1.placeholder(np.float32, name='input') # define the input tensor
    t_filter_weights = tf.compat.v1.placeholder(np.float32, name="filter_weights")
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    #create default graph
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
    print(layers)
  

 
    # Helper functions for TF Graph visualization
 
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        #plt.figure(figsize = (a.shape[0]/45,a.shape[1]/45))
        
        plt.imshow(a)
        plt.draw()
        #plt.show()
        plt.pause(0.00001)

        

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)
 
    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.compat.v1.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap
    
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.compat.v1.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)
    
    def resize_g(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(tf.compat.v1.image.resize_bilinear(img, size)))[0,:,:,:]
    resize_g = tffunc(np.float32, np.int32)(resize_g)
    
    def calc_grad_tiled(img, t_grad, filter_weights, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub, t_filter_weights:filter_weights})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)    


        
    #t_obj is the tensor for the layer selected for visualization 
    #img0 is image to be transformed - default noise image makes it easier to see patterns found by filter  
    #def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=8, octave_scale=1.4): #1st w/trump included
    def render_deepdream(t_obj, img0=img_noise, filt_img = [img_noise], iter_n=10, step=2.5, oor_step_scale=0.0, octave_n=26, octave_scale=1.2, octave_range=(0.0, 1.0), img0_weight=0.25, brightness=1.0): #2nd w/pug not included
        
        num_octaves = 0
        height = img0.shape[0]
        
        while height > 2:
            num_octaves += 1
            height = np.int32(np.float32(height)/octave_scale)
            
        layers = []    
        for obj,step_scale,iter_scale, color in t_obj:
            layers.append(obj)
            
            
        all_filter_weights, all_max_weights, all_layer_names = getAllActivations(layers, filt_img)
        
        grads = []
        for obj,step_scale,iter_scale, color in t_obj:
            #find layer index
            layer_index = -1
            for i in range(len(all_layer_names)):
                if(obj.name == all_layer_names[i]):
                    layer_index = i
                    break
                
            if layer_index == -1:
                print ("Layer name not found: " + obj.name)
                return
            
            #print(obj.shape.asl_list()[3])
            step_scale *= np.square(obj.shape.as_list()[3])/700000.0
            #print(step_scale)
            
            #filter_weights, max_weight = getActivations(obj, filt_img)
            filter_weights = all_filter_weights[i]
            max_weight = all_max_weights[i]
            print(obj.name)
            layer_scale = 1.0
            
            if "conv2d2" in obj.name:
                layer_scale = math.pow(max_weight/375, 4)
                
                
            if "conv2d1" in obj.name:
                layer_scale = math.pow(max_weight/325, 4)
                
            if "conv2d0" in obj.name:
                layer_scale = math.pow(max_weight/900, 4)
                
            if "mixed3a" in obj.name:
                layer_scale = math.pow(max_weight/200, 4)
                
            if "mixed3b" in obj.name:
                layer_scale = math.pow(max_weight/275, 4)
                
            if "mixed4a" in obj.name:
                layer_scale = math.pow(max_weight/265, 2)
            
            if "mixed4b" in obj.name:
                layer_scale = math.pow(max_weight/235, 2)
            
            if "mixed4c" in obj.name:
                layer_scale = math.pow(max_weight/105, 2)
                
            if "mixed4d" in obj.name:
                layer_scale = math.pow(max_weight/70, 2)
                
            if "mixed4e" in obj.name:
                layer_scale = math.pow(max_weight/110, 4)
                
            if "mixed5a" in obj.name:
                layer_scale = math.pow(max_weight/80, 4)
                
            if "mixed5b" in obj.name:
                layer_scale = math.pow(max_weight/30, 4)
            
            print(layer_scale)
            step_scale *= layer_scale
            
            obj = tf.square(obj)
            score = tf.reduce_mean(obj*t_filter_weights) # defining the optimization objective #obj is the square of the output of the layer - score is mse of layer output
            this_step = step_scale * step
            grads.append((tf.gradients(score, t_input)[0], this_step, iter_scale, color, filter_weights)) # behold the power of automatic differentiation!
    
        # split the image into a number of octaves
        img = img0
        
        octaves = []
        #for _ in range(octave_n-1):
        for _ in range(num_octaves-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            #print(img.shape)
            #print(resize(lo, hw).shape)
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
       
       ###########KLUDGE TEST 
        #img = img0
        
        # generate details octave by octave
        low_octave_index = np.int32(octave_range[0] * num_octaves)
        high_octave_index = np.int32(octave_range[1] * num_octaves-1)
        if high_octave_index == num_octaves:
            high_octave_index = num_octaves - 1
            
        for octave in range(num_octaves):
        #for octave in range(low_octave_index, high_octave_index):
            print('octave = ' + str(octave))
            print('low_octave_ind = ' + str(low_octave_index))
            print('high_octave_ind = ' + str(high_octave_index))
            if(octave < low_octave_index):
                continue
                
            if octave < low_octave_index or octave > high_octave_index:
                iter_num = 1
            else:
                iter_num = iter_n
  

            if octave>0:
                hi = octaves[-octave]
                print(hi.shape)
                if iter_num > 1:
                    img = (1-img0_weight)*resize(img, hi.shape[:2])+ img0_weight*resize(img0, hi.shape[:2]) + hi
                elif octave < low_octave_index:
                    img = resize_g(img, hi.shape[:2])
                else:
                    img = resize(img, hi.shape[:2])
            """
            for _ in range(iter_n):
                for grad in grads:
            """

            for grad in grads:
                grad_step = grad[1]
                num_iter = np.int32(grad[2]*iter_num)
                if octave < low_octave_index or octave > high_octave_index:
                    grad_step = grad_step * oor_step_scale
                    num_iter = 1
                    
                print(num_iter)
                for i in range(num_iter):
                    g = calc_grad_tiled(img, grad[0], grad[4])
                    if grad[3] == 0:
                        img += g*(grad_step / (np.abs(g).mean()+1e-7))
                    elif grad[3] == 1:
                        update = g*(grad_step / (np.abs(g).mean()+1e-7))
                        new_update = update
                        new_update[:,:,0] = update[:,:,0]
                        new_update[:,:,1] = update[:,:,2]
                        new_update[:,:,2] = update[:,:,1]
                        #print(new_update)
                        img += new_update
                        #img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                        #print(g*(grad[1] / (np.abs(g).mean()+1e-7)))
                    elif grad[3] == 2:
                        update = g*(grad_step/ (np.abs(g).mean()+1e-7))
                        new_update = update
                        new_update[:,:,0] = update[:,:,1]
                        new_update[:,:,1] = update[:,:,0]
                        new_update[:,:,2] = update[:,:,2]
                        #print(new_update)
                        img += new_update
                        #img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                        #print(g*(grad[1] / (np.abs(g).mean()+1e-7)))
                    elif grad[3] == 3:
                        update = g*(grad_step/ (np.abs(g).mean()+1e-7))
                        new_update = update
                        new_update[:,:,0] = update[:,:,1]
                        new_update[:,:,1] = update[:,:,2]
                        new_update[:,:,2] = update[:,:,0]
                        #print(new_update)
                        img += new_update
                        #img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                        #print(g*(grad[1] / (np.abs(g).mean()+1e-7)))
                    elif grad[3] == 4:

                        update = g*(grad_step / (np.abs(g).mean()+1e-7))
                        new_update = update
                        new_update[:,:,0] = update[:,:,2]
                        new_update[:,:,1] = update[:,:,0]
                        new_update[:,:,2] = update[:,:,1]
                        #print(new_update)
                        img += new_update
                        #img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                        #print(g*(grad[1] / (np.abs(g).mean()+1e-7)))
                    elif grad[3] == 5:
                        update = g*(grad_step / (np.abs(g).mean()+1e-7))
                        new_update = update
                        new_update[:,:,0] = update[:,:,2]
                        new_update[:,:,1] = update[:,:,1]
                        new_update[:,:,2] = update[:,:,0]
                        #print(new_update)
                        img += new_update
                        #img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                        #print(g*(grad[1] / (np.abs(g).mean()+1e-7)))
                        
                    if (i == num_iter-1):
                        
                        if brightness < 1.0:
                            img_out = img * brightness
                        elif brightness > 1.0:
                            #print(img)
                            inv_img = 255.0 - img
                            inv_img = inv_img*(brightness - 1.0)
                            img_out = img + inv_img
                            #print(inv_img)
                        else:
                            img_out = img
                            
                        showarray(img_out/255.0)
            
            #this will usually be like 3 or 4 octaves
            #Step 5 output deep dream image via matplotlib
            #showarray(img/255.0)
            print(octave)
        """    
        #do last full res  pass
        hi = octaves[-(num_octaves-1)]
        print(hi.shape)
        img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            for grad in grads:
                g = calc_grad_tiled(img, grad[0])
                img += g*(grad[1] / (np.abs(g).mean()+1e-7))
                if (i%8 == 0):
                        showarray(img/255.0)
        """
                
        #showarray(img/255.0)
        #showarray(((img-img0)+img0.mean())/255.0)
        
    def plotNNFilter(units):
        global plt
        filters = units.shape[3]
        plt.figure(1, figsize=(20,20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

            
    def getAllActivations(layers,images):
        filt_name = "dogface"
        all_filter_weights = []
        all_max_weights = []
        all_layer_names = []
        unfinished_layers = []
        for layer in layers:
            filename = "filter_weights/"+filt_name+"/" + layer.name + ".wgt"
            max_file = "filter_weights/"+filt_name+"/" + layer.name + ".max"
            if(os.path.isfile(filename)):
                filter_wts = np.fromfile(filename, dtype=np.float32)
                max_wt = np.fromfile(max_file, dtype=np.float32)
                print("Saved Max weight: " + str(max_wt))
                all_filter_weights.append(filter_wts)
                all_max_weights.append(max_wt)
                all_layer_names.append(layer.name)
                continue
            else:
                unfinished_layers.append(layer)
                
        all_units_names = []      
        all_units = []
        count = -1


        for image in images:
            #this_image_filter_weights = []
            #this_image_max_weights = []
            count += 1
            if(len(unfinished_layers) > 0):
                all_units = sess.run(unfinished_layers,feed_dict={t_input:image})
                
            for i in range(len(all_units)):
                all_units_names.append((unfinished_layers[i].name,all_units[i]))
            for name, units in all_units_names:
                filename = "filter_weights/"+filt_name+"/" + name + ".wgt.temp"+str(count)
                max_file = "filter_weights/"+filt_name+"/" + name + ".max.temp"+str(count)
                filters = units.shape[3]
                filter_weights = []
                max_weight = 0
                for i in range(filters):
                    unit = units[0, :, :, i]
                    #mean = tf.reduce_mean(unit)
                    mean_val = np.mean(unit)
                    """
                    if mean.eval() > 20:
                        filter_weights.append(1.0)
                    else:
                        filter_weights.append(0.0)
                        """
                    #mean_val = mean.eval()
                    if mean_val > max_weight:
                        max_weight = mean_val
                    filter_weights.append(mean_val)
                    
                #filter_weights /= max_weight
                
                """
                for i in range(len(filter_weights)):
                    if filter_weights[i] < 0.05:
                        filter_weights[i] = 0.0
                """
                """    
                units = units * filter_weights
                print(units.shape)
                print(np.array(filter_weights).shape)
                
                for i in range(filters):
                    unit = units[0, :, :, i]
                    mean = tf.reduce_mean(unit)
                    print(mean.eval())
                
                plotNNFilter(units)
                """
                #print("Max weight: " + str(max_weight))
                
                filter_weights = np.array(filter_weights)
                filter_weights.tofile(filename)
                max_weight = np.array(max_weight)
                max_weight.tofile(max_file)
                print("Saved " + filename)
                #all_filter_weights.append(filter_weights)
                #all_max_weights.append(max_weight)
                #all_layer_names.append(name)
        
        if len(unfinished_layers) > 0:
            for layer in layers:
                this_filter_weights = []
                this_max_weights = []
                for image_num in range(len(images)):
                    filename = "filter_weights/"+filt_name+"/" + layer.name + ".wgt.temp"+str(image_num)
                    max_file = "filter_weights/"+filt_name+"/" + layer.name + ".max.temp"+str(image_num)
                    
                    if(os.path.isfile(filename)):
                        filter_wts = np.fromfile(filename, dtype=np.float32)
                        max_wt = np.fromfile(max_file, dtype=np.float32)
                        print("Saved Max weight: " + str(max_wt))
                        
                        if len(this_filter_weights) == 0:
                            this_filter_weights = filter_wts
                        else:
                            this_filter_weights += filter_wts
                            
                        if len(this_max_weights) == 0:
                            this_max_weights = max_wt
                        else:
                            this_max_weights += max_wt
                            
                        #all_layer_names.append(layer.name)
                        os.remove(filename)
                        os.remove(max_file)
                        continue
                    
                new_filename = "filter_weights/"+filt_name+"/" + layer.name + ".wgt"
                new_max_file = "filter_weights/"+filt_name+"/" + layer.name + ".max"
                
                this_filter_weights /= this_max_weights
                this_filter_weights = np.array(this_filter_weights)
                this_filter_weights.tofile(new_filename)
                
                this_max_weights = np.array(this_max_weights)
                this_max_weights.tofile(new_max_file)
                
                all_filter_weights.append(this_filter_weights)
                all_max_weights.append(this_max_weights)
                all_layer_names.append(layer.name)
            
        return all_filter_weights, all_max_weights, all_layer_names
        
    def getActivations(layer,stimuli):
        filename = "filter_weights/fur/" + layer.name + ".wgt"
        max_file = "filter_weights/fur/" + layer.name + ".max"
        if(os.path.isfile(filename)):
            filter_wts = np.fromfile(filename, dtype=np.float32)
            max_wt = np.fromfile(max_file, dtype=np.float32)
            print("Max weight: " + str(max_wt))
            return filter_wts, max_wt
        
        units = sess.run(layer,feed_dict={t_input:stimuli})
        filters = units.shape[3]
        filter_weights = []
        max_weight = 0
        for i in range(filters):
            unit = units[0, :, :, i]
            mean = tf.reduce_mean(unit)
            """
            if mean.eval() > 20:
                filter_weights.append(1.0)
            else:
                filter_weights.append(0.0)
                """
            mean_val = mean.eval()
            if mean_val > max_weight:
                max_weight = mean_val
            filter_weights.append(mean_val)
            
        filter_weights /= max_weight
        
        """
        for i in range(len(filter_weights)):
            if filter_weights[i] < 0.05:
                filter_weights[i] = 0.0
        """
        """    
        units = units * filter_weights
        print(units.shape)
        print(np.array(filter_weights).shape)
        
        for i in range(filters):
            unit = units[0, :, :, i]
            mean = tf.reduce_mean(unit)
            print(mean.eval())
        
        plotNNFilter(units)
        """
        print("Max weight: " + str(max_weight))
        
        filter_weights = np.array(filter_weights)
        filter_weights.tofile(filename)
        max_weight = np.array(max_weight)
        max_weight.tofile(max_file)
        return filter_weights, max_weight
             
  
   	#Step 3 - Pick a layer to enhance our image
########## Conv
    l1 = 'conv2d2' #strokes
    l2 = 'conv2d1' #different strokes
    l3 = 'conv2d0' #neon edges

########## Mixed3  Inception Block?
    l4 = 'mixed3b_3x3_bottleneck_pre_relu'  #sweet 139 compartment swirls
    l5 = 'mixed3a_3x3_bottleneck_pre_relu'   #sweet 139 tiles
    l6 = 'mixed3a_5x5_bottleneck_pre_relu' # hazy orange fireflies sucks
    l7 = 'mixed3b_5x5_bottleneck_pre_relu'  #swirly tile

    l8 = 'mixed3b_5x5' #big swirl
    l9 = 'mixed3a_5x5' #fire

    l10 = 'mixed3b_3x3' #boiling
    l11 = 'mixed3a_3x3' #tile hieroglyphics
    

    l12 = 'mixed3b_1x1' # fire swirls cool
    l13 = 'mixed3b_1x1_pre_relu' # spots

    l14 = 'mixed3a_1x1' # detailed craggy
    l15 = 'mixed3a_1x1_pre_relu' #craggy

    l16 = 'mixed3a' #detailed tiled swirls nice
    l17 = 'mixed3b' #SWIRLY

    l18 = 'mixed3b_3x3_pre_relu/conv' #soft spots
    l19 = 'mixed3a_3x3_pre_relu/conv' #soft tiled
    l20 = 'mixed3b_5x5_pre_relu/conv' #extra swirly tile
    l21 = 'mixed3a_5x5_pre_relu/conv' #swirly hazy fireflies
    l22 = 'mixed3b_1x1_pre_relu/conv' #spots
    l23 = 'mixed3a_1x1_pre_relu/conv' #craggy

########## Mixed4
    l24 = 'mixed4d_3x3_bottleneck_pre_relu' # eyes maybe mouth
    l25 = 'mixed4a_3x3_bottleneck_pre_relu' #holes swirls
    l26 = 'mixed4b_3x3_bottleneck_pre_relu' #eyes swirls maybe fur
    l27 = 'mixed4c_3x3_bottleneck_pre_relu' #eyes and mouths dogfaces?
    
    l28 = 'mixed4d_5x5_bottleneck_pre_relu' # scary
    l29 = 'mixed4a_5x5_bottleneck_pre_relu' #swirly tile nice
    l30 = 'mixed4b_5x5_bottleneck_pre_relu' #doggy
    l31 = 'mixed4c_5x5_bottleneck_pre_relu' #crazy doggy

    l32 = 'mixed4d_3x3' #shaggy
    l33 = 'mixed4a_3x3' #eyeholes
    l34 = 'mixed4b_3x3' #eyefur scale background
    l35 = 'mixed4c_3x3' # gross mouths?

    l36 = 'mixed4d_5x5' #chicken skin flaps
    l37 = 'mixed4a_5x5' # tiled swirls cool
    l38 = 'mixed4b_5x5' #mouths dogface
    l39 = 'mixed4c_5x5' #furry dogfaces triangle patterns


    l40 = 'mixed4b_1x1' #dogface
    l41 = 'mixed4b_1x1_pre_relu' #pompom

    l42 = 'mixed4a_1x1' #swirly
    l43 = 'mixed4a_1x1_pre_relu' #swirly

    l44 = 'mixed4c_1x1' #animals
    l45 = 'mixed4c_1x1_pre_relu' #animal patterns

    l46 = 'mixed4d_1x1' #eyes scales feathers
    l47 = 'mixed4d_1x1_pre_relu' #dog eyes nose?

    l48 = 'mixed4a' #eyes furry
    l49 = 'mixed4b' #eyehole swirls
    l50 = 'mixed4c' #dogfaces landscape background
    l51 = 'mixed4d' #dog eye/nose

    l52 = 'mixed4b_3x3_pre_relu/conv'  #eyes furry
    l53 = 'mixed4a_3x3_pre_relu/conv'   #swirly holes
    l54 = 'mixed4c_3x3_pre_relu/conv'  #mouths
    l55 = 'mixed4d_3x3_pre_relu/conv'  #twisted flesh  #turtle plaid
    l56 = 'mixed4a_5x5_pre_relu/conv' #swoopy tiles
    l57 = 'mixed4b_5x5_pre_relu/conv' #eyes furry
    l58 = 'mixed4c_5x5_pre_relu/conv' #furry eyes
    l59 = 'mixed4d_5x5_pre_relu/conv' #mouths twisty flesh
    l60 = 'mixed4a_1x1_pre_relu/conv' #swirly tiles cool
    l61 = 'mixed4b_1x1_pre_relu/conv' # furry eyes
    l62 = 'mixed4c_1x1_pre_relu/conv' #furry eyes patterns
    l63 = 'mixed4c_1x1_pre_relu/conv' #furry dogfaces patterns
    
########## Mixed5 FREAKY
    l64 = 'mixed5b_3x3_bottleneck_pre_relu' #blackhole eyes bad skin  # birds
    l65 = 'mixed5a_3x3_bottleneck_pre_relu' #zombie punk #variety-lizard, scales, mouths, tech
    l66 = 'mixed5a_5x5_bottleneck_pre_relu' #whoa freaky #crazy.png
    l67 = 'mixed5b_5x5_bottleneck_pre_relu' #freaky cheek damage #chitin eyes hard flat

    l68 = 'mixed5b_5x5' #scaly skin #giger organic depth green
    l69 = 'mixed5a_5x5' #complex borg head mottled skin # tech nebula.png AWESOME

    l70 = 'mixed5b_3x3' #swirly texture skin colored hair #scaly cells closeup
    l71 = 'mixed5a_3x3'#colorful crazy eyes scales #ink drop swirls, feather mosaic
    

    l71 = 'mixed5b_1x1' #complex organic head #feather peacock spotted swooping
    l72 = 'mixed5b_1x1_pre_relu' #colorful plant #scales feathers
    
    l73 = 'mixed5a' #chicken/lizard face explosion #lizard face, insect wing
    l74 = 'mixed5b' #detailed yuck #crazy patterns: feather, brain, etc mosaic

    l75 = 'mixed5b_3x3_pre_relu/conv' #scary colorful eyes #chicken/lizard eye
    l76 = 'mixed5a_3x3_pre_relu/conv' #colorful zombie punk #lizard face/eye
    l77 = 'mixed5b_5x5_pre_relu/conv' #zombie eyes  #segmented chitin cells closeup
    l78 = 'mixed5a_5x5_pre_relu/conv' #mottled skin patterns #dogears, eyes
    l79 = 'mixed5b_1x1_pre_relu/conv' #feathery hole eyes #lizard/bird
    l80 = 'mixed5a_1x1_pre_relu/conv' #wavy fur eyes #tumors mouths


########## Head
    l81 = 'head1_bottleneck' #crazy borg detailed
    l82 = 'head0_bottleneck' #crazy scales eyes
    
    l83 = 'mixed4e'
    

    channel = 139 # picking some feature channel to visualize
    
    #imgM = PIL.Image.open('pilatus800.jpgmountain')
    imgM = PIL.Image.open('bulldog.jpg')
    size = imgM.size
    imgM = np.float32(imgM)
    
    '''
    #open image
    img0 = PIL.Image.open('pilatus800.jpg')
    #img0 = img0.resize(size)
    img0 = np.float32(img0)
    '''
    filter_images = []
    for file in os.listdir("filter_images"):
        if file.endswith(".jpg"):
            filter_images.append(PIL.Image.open("filter_images/"+file))

    
    
    plt.ion()
    plt.figure(figsize = (imgM.shape[0]/45,imgM.shape[1]/45))
    plt.show()

    #print([n.name for n in graph.as_graph_def().node])
    
    #layer_list = [l4,l7, l14, l69] #tunnel_weird.png #cat_cosmic
    #layer_list = [l1,l16, l69] #cat_weird.png
    layer_list = [l1,l14, l69] #mountain_weird.png
    #layer_list = [l12,l14, l69] #cat_fire.png
    #layer_list = [l1,l16, l77] #cat_blocks
    #layer_list = [l4, l77] #cat_chitin
    #layer_list = [l17, l69] #cat_swirl
    #layer_list = [l1,l16, l68] #cat_valley
    #layer_list = [l1,[l2,-0.25],l14, l69, [l3,-0.05]] #cat_sweet
    #layer_list = [l1, l2,[l3,0.2]]
    #layer_list = [l4,l69]
    
    #layer_list = [l1,[l2,-1.5,1],l14, l29, [l69,1,6], [l71,1,3], [l70,1,3], [l77,1,3],[l3,-0.05,1]] # cat_smooth iter-20, step=0.5 fullrange, scale-1.2
    #layer_list = [l1,l14, [l7,1,0.25], [l29,1,0.25],[l69,1,0.5], [l68,1,0.25], [l59,1,0.25]] #cat_nice i-10, s-1.0, fullrange, scale-1.1 #cat_detailed i-30, s=1.0 fullrange, scale 1.6
    #layer_list = [l4, [l16,1.5,0.25],[l7,1.5,0.25], [l29,1,0.25],[l69,5,1], [l68,1,0.25], [l59,1,0.25]]
    
    ##layer4
    
    #layer_list = [[l34,1,1,2]] #arches and animals in the distance
    #layer_list = [[l32,1,2.0,2]] #snake
    #layer_list = [[l29,1,1,2]] #swoopy tiles
    #layer_list = [[l27,1,2.0,2]] #sweet pattern iter-2:Jewelry?
    #layer_list = [[l25,1,2.0,2]] #swirl tile holes 
    #layer_list = [[l35,1,1.0,2]] #plaid snake bodies
    #layer_list = [[l37,1,1,2]] #tiled ribbons swirls
    #layer_list = [[l39,1,1,2]] #interesting baroque pattern
    #layer_list = [[l40,1,2,2]] #swirling snakeskin
    #layer_list = [[l41,1,1,2]] #radiating pattern
    #layer_list = [[l42,1,1,2]] #swirling holes, van goghish
    #layer_list = [[l43,1,1,2]] #pompom swirl tile
    #layer_list = [[l44,1,1,2]] #animals in background
    #layer_list = [[l46,1,2,2]] #lizard
    #layer_list = [[l47,1,1,2]] #swirly lizard
    #layer_list = [[l48,1,1,2]] #lizard
    #layer_list = [[l52,1,1,2]] #fur eye tile swirl
    #layer_list = [[l53,1,1,2]] #tight swirls
    #layer_list = [[l54,1,1,2]]# ornate silver
    #layer_list = [[l55,1,1,2]] #ornate arches
    #layer_list = [[l56,1,1,2]] #tiled ribbons
    #layer_list = [[l57,1,1,2]] #eyefur w background
    #layer_list = [[l59,1,1,2]] #hanging wrinkled cloth interesting
    #layer_list = [[l60,1,1,2]] #swoopy tiled swirls
    #layer_list = [[l61,1,1,2]] #eyefur
    #layer_list = [[l63,1,1,2]] # mixture eyefur, tiled swirls, scales, arches
    #liZards
    #layer_list = [[l66,1,1,2]] #patterns
    #layer_list = [[l66,1,1,0]] #cells mouth insanity
    #layer_list = [[l66,1,1,1]] 
    #layer_list = [[l66,1,1,3]] #neon barf
    #layer_list = [[l66,1,1,5],[l66,1,1,0]]
    #layer_list = [[l67,1,1,2]] #alligator skin pattern
    #layer_list = [[l67,1,1,0]] #insanity
    #layer_list = [[l70,1,1,0]] #chitin wings
    #layer_list = [[l78,1,2,0]] #interesting
    layer_list = [[l2,1,1,0],[l29,1,2,0],[l37,1,2,0],[l78,1,2,0],[l69,1,2,0],[l55,1,2,0]] #brussel sprouts
    #layer_list = [[l82,1,2,0]]

    #fur
    #layer_list = [[l1,0.02,0.3,0],[l2,0.02,0.3,0],[l3,0.02,0.3,0], [l16,0.5,0.5,0], [l17,0.5,0.5,0],[l48,1,0.2,0], [l49,1,0.2,0], [l50,1,0.2,0], [l51,1,0.2,0], [l83,0.2,0.2,0], [l73,0.01,0.5,0],[l74,0.01,0.5,0]]#[[l1,0.2,0.2,0], [l2,0.2,0.2,0], [l2,0.2, 0.2,0],[l48, 0.4, 0.4, 0], [l49, 0.4, 0.4, 0], [l50, 0.4, 0.4, 0], [l51, 0.4, 0.4, 0], l73, l74]
    layer_list = [[l1,1,0.3,0],[l2,1,0.3,0],[l3,1,0.3,0], [l16,1,0.5,0], [l17,1,0.5,0],[l48,1,0.2,0], [l49,1,0.2,0], [l50,1,0.2,0], [l51,1,0.2,0], [l83,1,0.2,0], [l73,1,0.5,0],[l74,1,0.5,0]]#[[l1,0.2,0.2,0], [l2,0.2,0.2,0], [l2,0.2, 0.2,0],[l48, 0.4, 0.4, 0], [l49, 0.4, 0.4, 0], [l50, 0.4, 0.4, 0], [l51, 0.4, 0.4, 0], l73, l74]
    layer_list = [[l73,1,0.5,0],[l74,1,0.5,0],[l48,1,0.2,0], [l49,1,0.2,0], [l50,1,0.2,0], [l51,1,0.2,0], [l83,1,0.2,0], [l16,1,0.5,0], [l17,1,0.5,0],[l1,1,0.3,0],[l2,1,0.3,0],[l3,1,0.3,0]]
    #snake
    #layer_list = [[l1,1,0.3,0],[l2,1,0.3,0],[l3,1,0.3,0], [l16,1,0.5,0], [l17,1,0.5,0],[l48,1,0.2,0], [l49,1,0.2,0], [l50,1,0.2,0], [l51,1,0.2,0], [l83,1,0.2,0], [l73,1,0.5,0],[l74,1,0.5,0]]#[[l1,0.2,0.2,0], [l2,0.2,0.2,0], [l2,0.2, 0.2,0],[l48, 0.4, 0.4, 0], [l49, 0.4, 0.4, 0], [l50, 0.4, 0.4, 0], [l51, 0.4, 0.4, 0], l73, l74]

    layer_list = [n if isinstance(n, (list, tuple)) else [n,1,1, 0] for n in layer_list]
    
    tf_layer_list = [[T(i[0]), i[1], i[2], i[3]] for i in layer_list]

    #Step 4 - Apply gradient ascent to that layer
    #render_deepdream(tf.square(T(layer)))
    #render_deepdream(tf.square(T('mixed4c')), img0)
    #render_deepdream([tf.square(T(layer4)), tf.square(T(layer69)), tf.square(T(layer81))], img0)
    #render_deepdream([tf.square(T(layer4)), tf.square(T(layer69)), tf.square(T(layer69))], img0)
    
    
    render_deepdream(tf_layer_list, imgM, filt_img=filter_images, iter_n=10, step=1.0, oor_step_scale=0.8, octave_range=(0.0, 1.0), octave_scale=1.6, img0_weight=0.001, brightness=1.1)
    #render_deepdream(tf_layer_list, imgM, filt_img=filter_images, iter_n=20, step=0.02, oor_step_scale=1.0, octave_range=(0.0, 1.0), octave_scale=1.4, img0_weight=0.001, brightness=1.1)
    
    #filter_weights = getActivations(T(l48), img0)
    #print(filter_weights)
      
    print("Done")
    # while True:
    #     plt.pause(0.1)  
  
if __name__ == '__main__':
    main()
