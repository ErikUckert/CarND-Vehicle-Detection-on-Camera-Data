import os
import glob
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
import sys
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import pickle

def read_images(vehicle_dir='vehicles'+os.sep,
                non_vehicle_dir = 'non-vehicles'+os.sep):

    #####
    ### make a list of all images to read in
    #####

    ## project folder contains a vehicles and a non-vehicles folder
    ## check the vehicles folder first

    ## set the right directory 
    image_types = os.listdir(vehicle_dir)
    os.path.join(vehicle_dir)
    cars = []

    ## read every image
    for image_type in image_types:
        cars.extend(glob.glob(vehicle_dir+image_type+'/*'))
        

    print('Number of cars found:', len(cars))

    with open('cars.txt', 'w') as file:
        for car in cars:
            file.write(car+'\n')

    ## read in the non-vehicle images

    ## set the right directory 
    #non_vehicle_dir = 'non-vehicles'+os.sep
    image_types = os.listdir(non_vehicle_dir)
    not_cars = []

    ## read every image
    for image_type in image_types:
        not_cars.extend(glob.glob(non_vehicle_dir+image_type+'/*'))

    print('Number of non-vehicle images found:', len(not_cars))

    with open('not_cars.txt', 'w') as file:
        for other in not_cars:
            file.write(other+'\n')

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    
    #####
    ### function to return HOG Features and visualize them
    #####

    ## call with two outputs if vis = True
    if vis == True:
        features, hog_image = hog(img, orientations = orient,
                                    pixels_per_cell = (pix_per_cell, pix_per_cell),
                                    cells_per_block = (cell_per_block, cell_per_block),
                                    transform_sqrt = False, visualize = vis,
                                    feature_vector = feature_vec)
        return features, hog_image
    
    ## otherwise call with one output
    else:
        features = hog(img, orientations = orient,
                                    pixels_per_cell = (pix_per_cell, pix_per_cell),
                                    cells_per_block = (cell_per_block, cell_per_block),
                                    transform_sqrt = False, visualize = vis,
                                    feature_vector = feature_vec)
        return features    

def bin_spatial(img, size=(32,32)):
    
    ##### 
    ### function is for "downsizing the image" to
    ### a spatial binarized feature vector
    ##### 

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
    
    #####
    ### function computes a color histogram vector
    #####

    channel1_hist = np.histogram(img[:,:,0], bins= nbins)
    channel2_hist = np.histogram(img[:,:,1], bins= nbins)
    channel3_hist = np.histogram(img[:,:,2], bins= nbins)

    ## concatenate the histogram into single feature vector
    hist_features = np.concatenate((channel1_hist[0],
                                    channel2_hist[0], 
                                    channel3_hist[0]))

    return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32,32),
                        hist_bins=32, orient=9,pix_per_cell=8,
                        cell_per_block=2, hog_channel=0,
                        spatial_feature=True, hist_feature=True,
                        hog_feature=True):
    
    #####
    ### functions uses the single feature extracting functions
    ### "get_hog_features", "bin_spatial" & "color_hist" to create
    ### a "all in one" feature vector
    ###
    ### This functions reads a whole list of images
    #####

    features = []

    for file in imgs:
        file_features = []
        image = mpimg.imread(file)

        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        
        if spatial_feature == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feature == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feature == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        ## append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))

    ## returning list of feature vectors
    return features

def training(list_of_cars, list_of_not_cars, n_samples, color_space,
                orient, pix_per_cell, cell_per_block, hog_channel,
                spatial_size, hist_bins, spatial_feature,
                hist_feature, hog_feature):

    #####
    ### function is for splitting the images in training an test
    ### data and train and test the classifier
    #####

    t = time.time()
    random_idxs = np.random.randint(0, len(list_of_cars), n_samples)
    test_cars = np.array(list_of_cars)[random_idxs]
    test_not_cars = np.array(list_of_not_cars)[random_idxs]

    car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient,pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feature=spatial_feature, hist_feature=hist_feature,
                            hog_feature=hog_feature)

    not_car_features = extract_features(test_not_cars, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient,pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feature=spatial_feature, hist_feature=hist_feature,
                            hog_feature=hog_feature)

    print(round(time.time()-t), 'Seconds to compute features...')

    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    ## fit a per column scaler
    X_scaler = StandardScaler().fit(X)

    ## apply the scaler to X
    scaled_X = X_scaler.transform(X)

    ## define the vector with the labels
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    ## split up the data into random training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,test_size=0.1,
                                                        random_state=rand_state)
    print('Using:', color_space, orient, 'orientations,',pix_per_cell, 'pixels per cell',
            cell_per_block, 'cells per block', hist_bins, 'histogram bins',
            spatial_size, 'spatial sampling')
    print('feature vector lenght:', len(X_train[0]))

    ## using a linear svc
    svc = LinearSVC()

    ## check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'seconds to train SVC')

    ## check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler

def find_cars(img, svc, X_scaler, y_start_stop, scale, color_space,
                                            orient, pix_per_cell, cell_per_block, hog_channel,
                                            spatial_size, hist_bins, spatial_feature,
                                            hist_feature, hog_feature):

    #####
    ### functions performs a slinding window search and predict
    ### vehicle position by rectangles
    #####

    img_boxes = []
    t = time.time()
    count = 0
    
    draw_img = np.copy(img)

    ## make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255

    ## crafting an image in terms of size and colorspace to search in
    img_to_search = img[y_start_stop[0]:y_start_stop[1],:,:]
    color_trans_to_search = color_conversion(img_to_search, conv='RGB2YCrCb')

    if scale != 1:
        imshape = color_trans_to_search.shape
        
        ## resizing the image with the scaling factor
        color_trans_to_search = cv2.resize(color_trans_to_search, (np.int(imshape[1]/scale),
                                            np.int(imshape[0]/scale)))
    ## setting the color channels
    ch1 = color_trans_to_search[:,:,0]
    ch2 = color_trans_to_search[:,:,1]
    ch3 = color_trans_to_search[:,:,2]

    ## define blocks & steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block **2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2 # this is instead of defining an overlap
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    ## compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False) 
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count +=1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            ## extract HOG features for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            ## extract the image path
            subimg = cv2.resize(color_trans_to_search[ytop:ytop+window, xleft:xleft+window], (64,64))

            ## get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            ## scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features,
                                                hist_features,
                                                hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                img_boxes.append(((xbox_left,ytop_draw+y_start_stop[0]),
                                    (xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0])))

    return img_boxes

def add_heat(heatmap, bbox_list):

    #####
    ## function is for creating a heatmap from list of
    ## car rectangles
    #####

    for box in bbox_list:
        ## Add += 1 for all pixels inside each bbox
        ## Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    ## Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):

    #####
    ## functions deletes low value heat
    #####

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    return heatmap

def draw_labeled_bboxes(img, labels):

    #####
    ## functions uses the label() function from scipy
    ## to create new boxes from the heatmap
    #####

    ## Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        ## Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        ## Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ## Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        ## Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    
    return img

def progress(count, total, status=''):
    
    #####
    ### Define a little progress bar for image processing
    ### from https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    #####
    
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

class detect_boxes():

    #####
    ### Defines a class to store boxes from video
    #####

    def __init__(self):
        self.previous_boxes = [] 
        
    def add_boxes(self, boxes):
        self.previous_boxes.append(boxes)

        if len(self.previous_boxes) > 20:
            self.previous_boxes = self.previous_boxes[len(self.previous_boxes)-20:]

                                            orient, pix_per_cell, cell_per_block, hog_channel,
                                            spatial_size, hist_bins, spatial_feature,
                                            hist_feature, hog_feature):

    #####
    ## function is for reading in an image stream
    ## processing it and write the new images
    ## to a new video
    #####

    ## set dimensions for output video
    frame_width = 1280
    frame_height = 720

    ## Define the codec and create VideoWriter object
    ## The output is stored in 'outpy.mp4' file.
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # defining the codec
    out = cv2.VideoWriter(output_video, fourcc, 20, (frame_width,frame_height))

    ## create a list of images from a video stream
    ## just for the progressbar
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0

    detected_boxes = detect_boxes()

    ## Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    counter = 0
    t = time.time()
    ## Read until video is completed
    while(cap.isOpened()):
        ret, image = cap.read() ## Capture frame-by-frame
        if ret == True:
            
            rects = []
            scales = (1.25, 1.5, 1.75, 2.0)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
            for scale in scales:
                rects.append(find_cars(img, svc, X_scaler, y_start_stop, scale, color_space = 'YCrCb',
                                                            orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                                                            spatial_size=(32,32), hist_bins=32, spatial_feature=True,
                                                            hist_feature=True, hog_feature=True))
                
            rectangles = [item for sublist in rects for item in sublist] 
                
            ## add detections to the history
            if len(rectangles) > 0:
                detected_boxes.add_boxes(rectangles) 

            rects = []

            heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
            for box in detected_boxes.previous_boxes:
                heatmap = add_heat(heatmap, box)
            heatmap = apply_threshold(heatmap, heat_treshhold)
            labels = label(heatmap)
            draw_img = draw_labeled_bboxes(np.copy(img), labels)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
            out.write(draw_img)

            ## display progressbar
            if frame_counter <= total_frames:
                progress(frame_counter, total_frames, status='Generating Video Output')
                frame_counter += 1

        ## Break the loop
        else: 
            break
    
    ## When everything done, release the video capture object
    cap.release()

    print('')
    print(time.time()-t, 'seconds to run')

def processing_pipeline(read_images=False, read_image_lists=False,
                        debug=False, train=False, load_classifier=False,
                        test_single_image=False, create_video=False, input_video='./test_video.mp4',
                        output_video='./test_out_video.mp4', n_samples = 1000, y_start_stop=[360,664],
                        scale=1.5, heat_treshhold=1, color_space = 'YUV',
                        orient = 9, pix_per_cell = 8, cell_per_block = 4, hog_channel = 'ALL',
                        spatial_size = (32,32), hist_bins = 16, spatial_feature = True,
                        hist_feature = True, hog_feature = True):

    if read_images == True:
        ## call this functions once to scan for images and 
        read_images()

    if read_image_lists == True:
        # read the .txt file containing the image path
        # and creating list of images for cars and not-cars
        with open('cars.txt') as file:
            list_of_cars = file.read().splitlines() # .splitlines to get rid of the \n

        with open('not_cars.txt') as file:
            list_of_not_cars = file.read().splitlines()

    if debug == True:
        ## call the feature debug function for visualizing single features
        feature_debug(list_of_cars, list_of_not_cars)

    if train == True:
        ## call the training function to train & test the classifier
        # color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
        # orients = [1,3,5,7,9,11]
        # pixels_per_cell = [4,8,16]
        # cells_per_block = [2,4,6,8,10]
        # spatial = [(8,8), (16,16), (32,32), (64,64)]
        # hist_bins = [8, 16, 32, 64, 128]
        # for i in range(len(hist_bins)):
        svc, X_scaler = training(list_of_cars, list_of_not_cars, n_samples, color_space,
                                            orient, pix_per_cell, cell_per_block, hog_channel,
                                            spatial_size, hist_bins, spatial_feature,
                                            hist_feature, hog_feature)


    
        
        joblib.dump(svc, 'svc.pkl')
        joblib.dump(X_scaler, 'X_scaler.pkl')

    if load_classifier == True:
        # loading a previously trained classifier
        svc = joblib.load('svc.pkl')
        X_scaler = joblib.load('X_scaler.pkl')

    if test_single_image == True:
        # call this function for testing the classifier on real images
        # by computing HOG and color features for every window
        searching_at_real_images(svc, X_scaler)

    if create_video == True:
        # call this funtion to create a video output
        create_video_output(svc, X_scaler, input_video, output_video, y_start_stop, scale, heat_treshhold,
                                            color_space, orient, pix_per_cell, cell_per_block, hog_channel,
                                            spatial_size, hist_bins, spatial_feature, hist_feature, hog_feature)

## calling the pipeline with the desired functions
processing_pipeline(read_images=False, read_image_lists=False,
                    debug=False, train=False, load_classifier=True,
                    test_single_image=False, create_video=True,
                    input_video='./input_images/project_video.mp4', output_video='./output_images/project_out_video.mp4',
                    n_samples=2000, y_start_stop=[400,520], scale=1, heat_treshhold=70, color_space = 'YCrCb',
                    orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 'ALL',
                    spatial_size = (32,32), hist_bins = 32, spatial_feature = True,
                    hist_feature = True, hog_feature = True)