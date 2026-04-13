from matplotlib import pyplot as plt
import scipy.io
import cv2
import numpy as np
import os
import pickle
import time
import multiprocessing as mp
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from scipy.spatial.distance import cdist

# Part 3.1

def featuresSURF(I, kp):
    """
    Compute local descriptors at externally supplied keypoints.

    Parameters
    ----------
    I : ndarray (H, W)
        Grayscale image.
    kp : iterable of (x, y, scale)
        Keypoints returned by the detector stage. Scale is converted to
        OpenCV's `size` convention (diameter in pixels).

    Returns
    -------
    des : ndarray (N, D)
        One descriptor vector per keypoint.
    """
    # I = I.astype(np.float32)/255
    # Convert custom keypoints (x, y, scale) to OpenCV keypoints.
    # OpenCV's `size` is the keypoint diameter, so we map scale to a local support window.
    kp_cv2 = [cv2.KeyPoint(x,y,2*np.ceil(3*scale)+1) for x,y,scale in kp]
    # SIFT descriptor extraction is used here for each provided keypoint location/size.
    fex = cv2.SIFT_create()
    _, des = fex.compute(I, kp_cv2)
    return des

def featuresHOG(I, points):
    """
    Compute one HOG descriptor per keypoint by cropping a local patch around each point.

    Notes
    -----
    The patch radius scales with keypoint scale (`rad = round(5 * scale)`), so larger
    keypoints produce descriptors from larger image neighborhoods.
    """
    # Normalize image intensities once so gradient magnitudes are scale-consistent.
    I = I.astype(float)/255
    N = points.shape[0]
    # Descriptor layout configuration: 2x2 cells, 9 orientation bins, 50% cell overlap.
    cells = np.array([2,2])
    nbins = 9
    cell_overlap = .5
    [h,w] = I.shape

    desc = []
    for i in range(N):
        # Points are stored as (x, y, scale); slicing uses (row=y, col=x).
        pi = int(points[i,1])
        pj = int(points[i,0])
        scale = points[i,2]
        # Use a scale-dependent square patch around each interest point.
        rad = int(round(5*scale))
        # Clamp boundaries so patches near image borders are still valid.
        img_patch = I[max(0,pi-rad):min(pi+rad,h)+1,max(0,pj-rad):min(pj+rad,w)+1]
        desc.append(simple_hog(img_patch,nbins,cells,cell_overlap))

    return np.float32(np.array(desc))

def simple_hog(Im,bins,cells,overlap=0.3,signed=0):
    """
    Compute a compact HOG descriptor from a single image patch.

    The patch is partitioned into an overlapping rectangular grid. For each cell:
    1) compute gradient orientation/magnitude per pixel,
    2) accumulate a magnitude-weighted orientation histogram,
    3) L2-normalize that histogram.
    Final output is the concatenation of all normalized cell histograms.
    """

    # % basic hog function
    # % Im : the input image
    # % bins : the number of bins
    # % cells : image segmentation. [cellsi cellsj] or one number for rectangular
    # % overlap : range : 0 - .5 for overlapping cells
    # % signed : 0 for unsigned , 1 for signed
    # % gradient_option : input to function imgradient.

    # Support both scalar-like `[n]` and explicit `[rows, cols]` cell specifications.
    if len(cells.shape) == 1:
        cellsi = cells[0]
        cellsj = cells[0]
    else:
        cellsi = cells[0]
        cellsj = cells[1]

    N, M = Im.shape
    # Use first-order image derivatives as local edge evidence.
    GY, GX = np.gradient(Im)
    magn, angle = cv2.cartToPolar(GX, GY)
    # Map negative angles to the target orientation range:
    # unsigned -> [0, pi), signed -> [0, 2*pi).
    angle[angle < 0] = (signed+1)*np.pi + angle[angle < 0]
    # Quantize orientation into `bins` equally spaced sectors.
    p = (signed+1)*np.pi/bins
    ind_angle = np.mod(np.floor(angle/p).astype(int), bins)

    # Build a rectangular cell grid and get each cell's half-size in pixels.
    P,patch_i,patch_j = rectangular_grid(N,M,cellsi,cellsj,overlap)

    local_desc = []
    for i in range(cellsi*cellsj):
        # Convert cell center + half-size to valid integer bounds.
        i_start = int(max(0,P[i,1]-patch_i))
        i_end = int(min(N-1,P[i,1]+patch_i))
        j_start = int(max(0,P[i,0]-patch_j))
        j_end = int(min(M-1,P[i,0]+patch_j))

        hist = np.zeros((1,bins))

        # Skip degenerate cells (possible for very small patches after border clipping).
        if (i_start != i_end and j_start != j_end):
            m_part = magn[i_start:i_end+1,j_start:j_end+1]
            a_part = ind_angle[i_start:i_end+1,j_start:j_end+1]
            mm = m_part.flatten()
            aa = a_part.flatten()
            # Weighted vote: each pixel contributes its gradient magnitude to its orientation bin.
            # `np.bincount` returns only up to max bin index present in this cell,
            # so we copy into the matching prefix of the fixed-length histogram.
            hist[0,0:max(aa)+1] = np.bincount(aa,weights=mm)

            # Cell-wise L2 normalization improves robustness to illumination/contrast changes.
            hist=hist/(np.linalg.norm(hist)+np.finfo(float).eps)

        local_desc.append(hist)

    # Final descriptor is the ordered concatenation of all local cell histograms.
    return np.concatenate(local_desc,axis=1).flatten()


def rectangular_grid(N,M,cellsi,cellsj,overlap):
    """
    Build an overlapping rectangular grid over an `N x M` patch.

    Returns
    -------
    P : ndarray (cellsi*cellsj, 2)
        Integer cell centers in `(x, y)` order.
    patch_i, patch_j : int
        Half-size of each cell support region in rows/cols.
    """
    # Solve for cell stride so `cellsi x cellsj` cells fit exactly with desired overlap.
    
    step_i = N/(cellsi*(1-overlap)+overlap)
    patch_i = round(step_i/2)
    step_j = M/(cellsj*(1-overlap)+overlap)
    patch_j = round(step_j/2)

    xr = np.linspace(patch_j,M-1-patch_j,cellsj)
    yr = np.linspace(patch_i,N-1-patch_i,cellsi)

    X, Y = np.meshgrid(xr,yr)
    # P contains integer cell centers as (x, y) pairs.
    P = np.concatenate([X.reshape((-1,1)), Y.reshape((-1,1))], axis=1).round().astype(int)
    return (P, patch_i, patch_j)

# Part 3.2

def extract_feature_sets(detector_fun, descriptor_fun, loadFile=None, saveFile=None, distort=False):
    """
    Extract local descriptors for each image in each class.

    Workflow
    --------
    1) Load class images from `./Data/<class_folder>`.
    2) Convert to grayscale, optionally apply distortion, downsample by 0.5.
    3) Detect keypoints and compute descriptors per image.
    4) Return nested structure: `X_full[class_idx][image_idx] = descriptors`.

    - X_full is the full extracted-features container for the whole dataset.
    - X_full[class_idx][image_idx] -> descriptor matrix for one image

    Parameters
    ----------
    detector_fun : callable
        Function mapping a normalized grayscale image to keypoints.
    descriptor_fun : callable
        Function mapping `(image, keypoints)` to descriptor matrix.
    loadFile : str or None
        If provided, skips extraction and loads precomputed `X_full`.
    saveFile : str or None
        If provided, stores computed `X_full` for reuse.
    distort : bool
        Whether to apply `distort_image` as data augmentation.
    """

    # Computational Efficiency: Processing high-resolution images is computationally expensive.
    # Reducing the image size significantly speeds up the detection of interest points and the
    # calculation of descriptors across the entire dataset.




    #Keypoints are distinctive image locations (like corners, blobs, or textured spots) that
    # are stable enough to be found again under changes in scale, rotation, or lighting.
    #They act as anchor points where you compute descriptors for matching or recognition.

    if loadFile is not None:
        # Fast path: reuse precomputed descriptors from disk.
        X_full = pickle.load(open(loadFile,'rb'))
        return X_full

    # Class label name + corresponding folder name on disk.
    # Their order defines the numeric class ids used later.
    dataDir = './Data'
    categories = [
        ('person','persons'),
        ('cars','cars'),
        ('bike','bikes')
    ]

    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_procs = 1
    else:
        num_procs = min(3,num_cpus-1)
    # `num_procs` is kept for compatibility; extraction below currently runs serially.
    # If parallelized later, this value is already prepared.

    start_time = time.time()
    im_full = []
    for name, catDir in categories:
        img_list = sorted(os.listdir(os.path.join(dataDir, catDir)))
        im_class = []
        count = 0
        for img_file in tqdm(img_list, total=len(img_list), desc=f"Reading {name} images"): #purely for iteration with progress bar
            # Filenames may contain extra assets; keep only images for this category.
            if name not in img_file:
                continue
            # Load grayscale image, optionally apply synthetic distortion, then downsample.
            I = cv2.cvtColor(cv2.imread(os.path.join(dataDir, catDir, img_file)), cv2.COLOR_BGR2GRAY)
            if distort:
                I = distort_image(I)
            # cv2.INTER_AREA uses area-based resampling, which is usually best for
            # shrinking images because it reduces aliasing/noise and gives smoother results.
            I = cv2.resize(I, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            im_class.append(I)

        # Store one tuple per class to keep detector/descriptor configuration explicit.
        # tupple: [(detector_fun, descriptor_fun, im_class)]
        # And im_class itself is the list of images for that class.
        im_full.append((detector_fun,descriptor_fun,im_class))



    # print('Time for reading all images from disk: {:.3f}'.format(time.time()-start_time))

    start_time = time.time()

    # Each class is processed independently; output preserves class order in `categories`.
    X_full = list(map(extract_feature_batch, im_full))

    print('Time for feature extraction: {:.3f}'.format(time.time()-start_time))

    if saveFile is not None:
        pickle.dump(X_full, open(saveFile,'wb'))

    return X_full


def extract_feature_batch(tup):
    """
    Process one class worth of images.

    Input tuple format:
    `(detector_function, descriptor_function, list_of_images)`.
    """
    detect_fun, desc_fun, im_class = tup
    X_class = []
    for I in im_class:
        # Detector runs on normalized intensities; descriptor consumes raw grayscale + keypoints.
        kp = detect_fun(I.astype(float)/255)
        descriptors = desc_fun(I,kp)
        X_class.append(descriptors)
    
    return X_class


def create_train_test_split(features, k=None):
    """
    Build train/test splits using predefined fold indices from `fold_indices.mat`.

    Parameters
    ----------
    features : list
        Nested feature list returned by `FeatureExtraction`.
    k : int
        Fold index used to select a predefined permutation/split.

    Returns
    -------
    data_train, label_train, data_test, label_test : tuple
        Flattened descriptor sets and aligned integer class labels.
    """
    r = 0.7 #70% for training and 30% for testing.
    if k is not None:
        # Predefined folds guarantee reproducible splits across experiments.
        mat = scipy.io.loadmat('Fold_Indices.mat')
        indices = mat['Indices'].flatten()[k].flatten()
    else:
        raise ValueError('createTrainTest: Please provide fold index (1-5).')

    data_train = []
    data_test = []
    label_train = []
    label_test = []
    for c in range(indices.shape[0]):
        idx_class = indices[c].flatten()
        #.flatten(): converts it to a 1D array.
        #Example: [[2], [5], [7]] -> [2, 5, 7].
        # Reorder class features according to fold definition before splitting.
        feats_class = [features[c][i] for i in idx_class] #In Python, specifically when using libraries like NumPy,
        # OpenCV, or PyTorch, .shape is an attribute (not a function, so you don't use parentheses) that tells
        # you the dimensions of an array or image.

        lim = int(round(r*idx_class.shape[0])) #Since idx_class was flattened,
        # it’s 1D, so this is just the number of elements in it (its length).

        data_train.extend(feats_class[:lim])
        #a = [1, 2]
        #a.extend([3, 4]) -> [1, 2, 3, 4]

        data_test.extend(feats_class[lim:])
        #feats_class[:lim] includes indices 0 through lim-1
        label_train.extend([c for i in range(len(feats_class[:lim]))])
        label_test.extend([c for i in range(len(feats_class[lim:]))])

    return (data_train, label_train, data_test, label_test)

def build_bag_of_words(data_train, data_test):
    """
    Convert variable-length local descriptor sets into fixed-length BoW histograms.

    Steps
    -----
    1) Learn a visual vocabulary with k-means on pooled train descriptors.
    2) Assign each descriptor to its nearest cluster center (hard assignment).
    3) Build one normalized histogram per image over visual words.
    """


    # Use only this fraction of pooled train descriptors for k-means fitting (speed/accuracy tradeoff).
    slice_ratio = 0.5
    # Number of visual words (k-means clusters), i.e., BoW histogram dimensionality.
    num_centers = 500

    # Build one large descriptor pool from all training images for vocabulary learning.
    train_all = np.concatenate(data_train, axis = 0)
    np.random.shuffle(train_all)
    # print(train_all.shape)

    # `n_init=1` keeps runtime low for lab settings; higher values usually improve stability.
    clf = KMeans(n_clusters = num_centers, n_init=1, verbose=False)

    # Learn visual words from a random subset to reduce k-means cost.
    clf.fit(train_all[:round(slice_ratio*train_all.shape[0])])
    C = clf.cluster_centers_
    BOF_tr = np.zeros((len(data_train), num_centers)) #train
    BOF_ts = np.zeros((len(data_test), num_centers)) #test

    for img_idx in range(len(data_train)):
        # Hard assignment: each local descriptor votes for its nearest visual word.
        closest = np.argmin(cdist(data_train[img_idx], C), axis=1)
        hist = np.zeros(num_centers)
        # Fill only the valid prefix returned by `bincount`; untouched bins remain zero.
        hist[0:np.max(closest)+1] = np.bincount(closest)
        # L2-normalized histogram is the global image representation.
        BOF_tr[img_idx,:] = hist/(np.linalg.norm(hist) + np.finfo(float).eps)

    for img_idx in range(len(data_test)):
        closest = np.argmin(cdist(data_test[img_idx], C), axis=1)
        hist = np.zeros(num_centers)
        hist[0:np.max(closest)+1] = np.bincount(closest)
        BOF_ts[img_idx,:] = hist/(np.linalg.norm(hist) + np.finfo(float).eps)

    return (BOF_tr, BOF_ts)

def svm(train_data, train_labels, test_data, test_labels, cost=1, svm_type='linear'):
    """
    Train and evaluate a multiclass SVM via one-vs-rest decomposition.

    Returns
    -------
    acc : float
        Top-1 classification accuracy on `test_data`.
    predictions : ndarray
        Predicted class id for each test sample.
    probs : ndarray
        Per-class probability estimates from the underlying SVMs.
    """

    # acc == accuracy

    # SVM:
    #An SVM is a model that draws a boundary to separate categories.
    #For two classes, think:
    #points of class A on one side
    #points of class B on the other side
    #It picks the boundary that separates them with the biggest gap (margin), which usually helps generalization.

    # One-vs-Rest: train one binary SVM per class (class c vs all others),
    # Example: If you have 3 classes (person, car, bike), you train 3 separate yes/no models
    # Check all class models and pick the class that says "yes" most strongly. (assuming the svm selects between (yes/no) car)
    # A class is the target label/category (ground truth), like person, car, bike
    #A binary SVM is an SVM that classifies between exactly two classes.


    # Number of classes inferred from labels; retained for readability/debug checks.
    num_classes = len(np.unique(train_labels))

    if svm_type == 'chi2':
        raise NotImplementedError()

    if svm_type == 'linear':
        # Binary base classifier used by One-vs-Rest for multiclass classification.
        base_clf = SVC(C=cost, kernel='linear', probability=True,  verbose=0)
        # SVC is scikit-learn's Support Vector Classifier (an SVM model) used for classification.
    elif svm_type == 'chi2':
        raise NotImplementedError()
    else:
        raise ValueError('svm: Unsupported Kernel Type.')

    # Wrap binary SVM to train one classifier per class against all others.
    # It takes your binary SVM (base_clf) and turns it into a multiclass model
    # by creating one binary classifier for each class (class c vs all other classes).

    #Example:
    #Binary: car vs not car
    #Multiclass: choose one from car, bike, person

    multisvm = OneVsRestClassifier(base_clf)

    multisvm.fit(train_data, train_labels)

    predictions = multisvm.predict(test_data)
    # Class probabilities are useful for ROC/PR analysis beyond top-1 predictions.
    probs = multisvm.predict_proba(test_data)

    acc = np.sum(predictions == test_labels)/len(test_labels)

    return (acc, predictions, probs)

