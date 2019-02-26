def load_numeric_training(standardize=True):
    '''
    Loads pre-extracted features for the training data
    -------------returns-------------
    ID: Image ids
    X: Data
    y: label
    '''
    # read from csv file
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    # extracting the labels
    y = data.pop('species')

    # label encoding the target `species`
    leaf_encoder = LabelEncoder()
    y = leaf_encoder.fit(y).transform(y)

    if standardize:
        # standardizing the data
        leaf_scaler = StandardScaler()
        X = leaf_scaler.fit(data).transform(data)
    else:
        X = data.values
        
    return ID, X, y    

def resize_img(img, max_dim=96):
    '''
    Resize the image so that max side is size `max_dim`
    -------------Arguements-------------
    img(PIL.image): loads raw image from Keras.preprocessing.image
    max_dim: maximum image dimension to resize image into
    -------------returns-------------
    image(np.array): Resized image with largest dim as `max_dim`
    '''
    # get axis with largest size 
    max_axis = max((0, 1), key=lambda i: img.size[i])

    # scaling both the axis to make the largest dim as `max_axis`
    scale = max_dim / img.size[max_axis]
    
    return img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))

def load_image_data(ids, max_dim=96, center=True):
    '''
    Takes input array of image IDs &
    returns resized image
    with largest dim as `max_dim`
    -------------Arguements-------------
    ids: IDs of images 
    max_dim: maximum image dimension to resize image into
    center: place image in the center
    -------------returns-------------
    `np.array` type image matrix with padding
        if image max(dim) < max_dim
    '''
    # init o/p array
    X = np.empty((len(ids), max_dim, max_dim, 1))
    
    for i, idee in enumerate(ids):
        # cast image into np.array
        x = resize_img(load_img(os.path.join(
                                    root, 'images', str(idee)+'.jpg'),
                                    grayscale=True),
                                max_dim = max_dim)
        # image to array
        x = img_to_array(x)
        
        # get corners of boundingbox
        length = x.shape[0]
        width = x.shape[1]
        
        if center: # center place image
            h1 = int((max_dim - length)/2)
            h2 = h1 + length
            w1 = int((max_dim - width)/2)
            w2 = w1 + width
            
        else: # top-left corner
            h1, w1 = 0, 0
            h2, w2 = (length, width)
            
        # insert image into Image matrix
        X[i, h1:h2, w1:w2, 0:1] = x
        
    return np.around(X / 255.0)

def load_train_data(split=split, random_state=None):
    '''
    load the pre-extracted features and image data.
    Split them into train-test and returns them
    -------------Arguments-------------
    split(float)(0-1): train split size
    random_state(num): random seed state
    -------------returns-------------
    train data tuple: (data features(np.array), image data(np.array), target values(np.array))
    test data tuple: (data features(np.array), image data(np.array), target values(np.array))
    '''
    # loading pre-extracted features
    ID, X_num, y = load_numeric_training()

    # load image data
    X_img = load_image_data(ID)

    # split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split,
                                random_state=random_state)

    # cast all data(X_num, y) into a generator function
    # & returns the train-test split indexes
    train_ind, test_ind = next(sss.split(X_num, y))

    # loading test indices
    X_num_val_te, X_img_te, y_val_te = X_num[test_ind], X_img[test_ind], y[test_ind]

    # loading train indices
    X_num_val_tr, X_img_tr, y_val_tr = X_num[train_ind], X_img[train_ind], y[train_ind]

    return (X_num_val_tr, X_img_tr, y_val_tr), (X_num_val_te, X_img_te, y_val_te)